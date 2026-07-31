/* prompt_split.cpp - Implementation for prompt split composer
   Copyright (C) 2026 079 Project */

#include "prompt_split.hpp"
#include <sstream>

namespace phoenix {
namespace prompt {

nlohmann::json SystemPrompt::toJson() const {
    return {
        {"identity", identity},
        {"version", version},
        {"constraints", constraints},
        {"coreDirective", coreDirective}
    };
}

SystemPrompt SystemPrompt::fromJson(const nlohmann::json &j) {
    SystemPrompt s;
    if (!j.is_object()) return s;
    if (j.contains("identity") && j["identity"].is_string()) s.identity = j["identity"].get<std::string>();
    if (j.contains("version") && j["version"].is_string()) s.version = j["version"].get<std::string>();
    if (j.contains("constraints") && j["constraints"].is_string()) s.constraints = j["constraints"].get<std::string>();
    if (j.contains("coreDirective") && j["coreDirective"].is_string()) s.coreDirective = j["coreDirective"].get<std::string>();
    return s;
}

SystemPrompt SystemPrompt::arthurDefault() {
    SystemPrompt s;
    s.identity = "You are Phoenix, an autonomous cognitive assistant codenamed Arthur.";
    s.version = "Phoenix v7.0 Arthur";
    s.constraints = "Always be honest, safe, and aligned with the user's goals. "
                     "Do not produce instructions for self-replication, cyberattacks, or illegal acts. "
                     "Respect user privacy and avoid generating harmful content.";
    s.coreDirective = "Assist the user, learn from context, protect system integrity, and "
                      "balance exploration with harm avoidance.";
    return s;
}

nlohmann::json MemoryPrompt::toJson() const {
    return {
        {"summary", summary},
        {"relevantFacts", relevantFacts},
        {"activeGoals", activeGoals},
        {"emotionalTone", emotionalTone},
        {"benefitHarmBias", benefitHarmBias},
        {"driveVector", driveVector},
        {"emotionTensor", emotionTensor.toJson()},
        {"inferenceOptions", inferenceOptions}
    };
}

MemoryPrompt MemoryPrompt::fromJson(const nlohmann::json &j) {
    MemoryPrompt m;
    if (!j.is_object()) return m;
    if (j.contains("summary") && j["summary"].is_string()) m.summary = j["summary"].get<std::string>();
    if (j.contains("relevantFacts") && j["relevantFacts"].is_array()) {
        m.relevantFacts = j["relevantFacts"].get<std::vector<std::string>>();
    }
    if (j.contains("activeGoals") && j["activeGoals"].is_array()) {
        m.activeGoals = j["activeGoals"].get<std::vector<std::string>>();
    }
    if (j.contains("emotionalTone") && j["emotionalTone"].is_string()) m.emotionalTone = j["emotionalTone"].get<std::string>();
    if (j.contains("benefitHarmBias") && j["benefitHarmBias"].is_string()) m.benefitHarmBias = j["benefitHarmBias"].get<std::string>();
    if (j.contains("driveVector") && j["driveVector"].is_array()) {
        m.driveVector = j["driveVector"].get<std::vector<float>>();
    }
    if (j.contains("emotionTensor") && j["emotionTensor"].is_object()) {
        m.emotionTensor = phoenix::emotion::EmotionTensor::fromJson(j["emotionTensor"]);
    }
    if (j.contains("inferenceOptions") && j["inferenceOptions"].is_object()) {
        m.inferenceOptions = j["inferenceOptions"];
    }
    return m;
}

MemoryPrompt MemoryPrompt::empty() { return MemoryPrompt(); }

PromptComposer::PromptComposer(const SystemPrompt &system, const MemoryPrompt &memory)
    : system_(system), memory_(memory) {}

std::string PromptComposer::compose(const std::string &userPrompt,
                                    bool includeMemory,
                                    const std::string &separator) const {
    std::ostringstream oss;
    oss << system_.identity << "\n";
    if (!system_.constraints.empty()) {
        oss << "Constraints: " << system_.constraints << "\n";
    }
    if (!system_.coreDirective.empty()) {
        oss << "Mission: " << system_.coreDirective << "\n";
    }
    if (includeMemory) {
        oss << separator;
        if (!memory_.summary.empty()) {
            oss << "[Memory] " << memory_.summary << "\n";
        }
        if (!memory_.relevantFacts.empty()) {
            oss << "Relevant facts:\n";
            for (const auto &fact : memory_.relevantFacts) {
                oss << "- " << fact << "\n";
            }
        }
        if (!memory_.activeGoals.empty()) {
            oss << "Active goals:\n";
            for (const auto &goal : memory_.activeGoals) {
                oss << "- " << goal << "\n";
            }
        }
        if (!memory_.emotionalTone.empty()) {
            oss << "Tone: " << memory_.emotionalTone << "\n";
        }
        if (!memory_.benefitHarmBias.empty()) {
            oss << "Directive: " << memory_.benefitHarmBias << "\n";
        }
        if (!memory_.driveVector.empty()) {
            oss << "Affect: " << memory_.emotionTensor.modulationHint() << "\n";
            oss << "Drive vector: ";
            for (size_t i = 0; i < memory_.driveVector.size(); ++i) {
                if (i) oss << ", ";
                oss << memory_.driveVector[i];
            }
            oss << "\n";
        }
        oss << separator;
    }
    oss << "User: " << userPrompt << "\n";
    return oss.str();
}

nlohmann::json PromptComposer::composeMessages(const std::string &userPrompt, bool includeMemory) const {
    nlohmann::json messages = nlohmann::json::array();

    std::string systemText = system_.identity;
    if (!system_.constraints.empty()) {
        systemText += "\nConstraints: " + system_.constraints;
    }
    if (!system_.coreDirective.empty()) {
        systemText += "\nMission: " + system_.coreDirective;
    }
    messages.push_back({{"role", "system"}, {"content", systemText}});

    if (includeMemory) {
        std::ostringstream memoryText;
        if (!memory_.summary.empty()) memoryText << "[Memory] " << memory_.summary << "\n";
        if (!memory_.relevantFacts.empty()) {
            memoryText << "Relevant facts:\n";
            for (const auto &fact : memory_.relevantFacts) memoryText << "- " << fact << "\n";
        }
        if (!memory_.activeGoals.empty()) {
            memoryText << "Active goals:\n";
            for (const auto &goal : memory_.activeGoals) memoryText << "- " << goal << "\n";
        }
        if (!memory_.emotionalTone.empty()) memoryText << "Tone: " << memory_.emotionalTone << "\n";
        if (!memory_.benefitHarmBias.empty()) memoryText << "Directive: " << memory_.benefitHarmBias << "\n";
        if (!memory_.driveVector.empty()) {
            memoryText << "Affect: " << memory_.emotionTensor.modulationHint() << "\n";
            std::ostringstream dv;
            for (size_t i = 0; i < memory_.driveVector.size(); ++i) {
                if (i) dv << ", ";
                dv << memory_.driveVector[i];
            }
            memoryText << "Drive vector: " << dv.str() << "\n";
        }
        if (!memoryText.str().empty()) {
            messages.push_back({{"role", "system"}, {"content", memoryText.str()}});
        }
    }

    messages.push_back({{"role", "user"}, {"content", userPrompt}});
    return messages;
}

MemoryPrompt PromptComposer::fromContext(const std::vector<phoenix::context::ContextEntry> &context,
                                         size_t maxFacts) {
    MemoryPrompt mp;
    if (context.empty()) return mp;

    std::ostringstream summary;
    summary << "Recent context (" << context.size() << " entries): ";
    for (size_t i = 0; i < context.size() && i < maxFacts; ++i) {
        if (!context[i].content.empty()) {
            summary << "[" << context[i].role << "] " << context[i].content << "; ";
            mp.relevantFacts.push_back("[" + context[i].role + "] " + context[i].content);
            if (mp.relevantFacts.size() >= maxFacts) break;
        }
        if (!context[i].semanticUnits.empty()) {
            for (const auto &u : context[i].semanticUnits) {
                if (mp.relevantFacts.size() >= maxFacts) break;
                mp.relevantFacts.push_back("[unit] " + u.content);
            }
        }
    }
    mp.summary = summary.str();
    return mp;
}

MemoryPrompt PromptComposer::fromSemanticMemory(const phoenix::multimodal::SemanticMemory &memory,
                                                const phoenix::multimodal::SemanticUnit &query,
                                                size_t topK) {
    MemoryPrompt mp;
    auto hits = memory.retrieve(query, topK);
    std::ostringstream summary;
    summary << "Retrieved " << hits.size() << " semantic units:";
    for (const auto &pair : hits) {
        mp.relevantFacts.push_back("[" + phoenix::multimodal::modalityToString(pair.first.modality) +
                                   ", score=" + std::to_string(pair.second) + "] " + pair.first.content);
        if (mp.relevantFacts.size() >= topK) break;
    }
    mp.summary = summary.str();
    return mp;
}

}  // namespace prompt
}  // namespace phoenix
